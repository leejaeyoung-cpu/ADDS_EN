"""
Configuration loader utility for ADDS
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigLoader:
    """
    Centralized configuration management for ADDS
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config.yaml. If None, uses default
        
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Expand environment variables in paths
        self._expand_env_vars()
        
        return self._config
    
    def _expand_env_vars(self):
        """Replace environment variable references in config"""
        if 'paths' in self._config:
            for key, value in self._config['paths'].items():
                if isinstance(value, str) and value.startswith('$'):
                    env_var = value[1:]
                    self._config['paths'][key] = os.getenv(env_var, value)
        
        # Apply environment variable defaults for CUDA settings
        self._apply_env_defaults()
    
    def _apply_env_defaults(self):
        """
        Apply environment variable defaults for CUDA configuration
        These serve as defaults and can be overridden at runtime by UI or explicit parameters
        
        Environment Variables:
            ADDS_DEVICE: 'cuda' or 'cpu' - Default training device
            ADDS_CELLPOSE_GPU: 'true' or 'false' - Default Cellpose GPU setting
        """
        # Override training device
        device_override = os.getenv('ADDS_DEVICE')
        if device_override:
            if 'training' not in self._config:
                self._config['training'] = {}
            self._config['training']['device'] = device_override.lower()
            print(f"[CONFIG] Device default from environment: {device_override}")
        
        # Override Cellpose GPU setting
        cellpose_gpu_override = os.getenv('ADDS_CELLPOSE_GPU')
        if cellpose_gpu_override:
            if 'cellpose' not in self._config:
                self._config['cellpose'] = {}
            self._config['cellpose']['gpu'] = cellpose_gpu_override.lower() in ['true', '1', 'yes']
            print(f"[CONFIG] Cellpose GPU default from environment: {cellpose_gpu_override}")

    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
        
        Returns:
            Configuration value
        
        Example:
            config = ConfigLoader()
            db_host = config.get('database.host')
            model_lr = config.get('models.gnn.learning_rate', 0.001)
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-notation key
        
        Args:
            key: Configuration key
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """
        Save current configuration to file
        
        Args:
            output_path: Output file path. If None, overwrites original
        """
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ Configuration saved to {output_path}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        return self._config


# Global config instance
config = ConfigLoader()
