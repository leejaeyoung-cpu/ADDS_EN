"""
Configuration management for ADDS Backend
Environment-based settings using Pydantic BaseSettings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_title: str = "ADDS API"
    api_version: str = "1.0.0"
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        env="CORS_ORIGINS"
    )
    
    # Database
    database_url: str = Field(
        default="sqlite:///data/analysis_results.db",
        env="DATABASE_URL"
    )
    
    # Models
    model_cache_dir: Path = Field(
        default=Path("models"),
        env="MODEL_CACHE_DIR"
    )
    cellpose_model_type: str = Field(default="cyto2", env="CELLPOSE_MODEL_TYPE")
    
    # GPU Configuration
    gpu_enabled: bool = Field(default=False, env="GPU_ENABLED")
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    
    # Upload Limits
    max_upload_size: int = Field(default=100 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 100MB
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="text", env="LOG_FORMAT")  # text or json
    log_file: Optional[Path] = Field(default=Path("logs/api.log"), env="LOG_FILE")
    
    # Redis (for caching - optional)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Security
    api_key_enabled: bool = Field(default=False, env="API_KEY_ENABLED")
    api_keys: List[str] = Field(default=[], env="API_KEYS")
    
    # Performance
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")  # development, staging, production
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            Path("data"),
            Path("data/outputs"),
            Path("data/raw"),
            Path("data/processed"),
            self.model_cache_dir,
            Path("logs")
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Setup directories on import
settings.setup_directories()
