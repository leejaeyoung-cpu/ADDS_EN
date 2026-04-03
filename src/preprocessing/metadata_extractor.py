"""
Advanced Image Metadata Extractor for ADDS
자동으로 이미지의 상세한 메타데이터를 추출하고 분석 정보를 저장
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import yaml
from PIL import Image
from PIL.ExifTags import TAGS
import cv2

from utils import get_logger

logger = get_logger(__name__)


class ImageMetadataExtractor:
    """
    이미지로부터 상세한 메타데이터를 추출하고 분석 정보를 생성
    """
    
    def __init__(self):
        """Initialize metadata extractor"""
        logger.info("ImageMetadataExtractor initialized")
    
    def extract_exif_data(self, image_path: Path) -> Dict[str, Any]:
        """
        EXIF 데이터 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            EXIF 데이터 딕셔너리
        """
        try:
            image = Image.open(image_path)
            exif_data = {}
            
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value) if not isinstance(value, (int, float, str)) else value
            
            return exif_data
        except Exception as e:
            logger.warning(f"Failed to extract EXIF data: {e}")
            return {}
    
    def extract_basic_info(self, image_path: Path, image: np.ndarray) -> Dict[str, Any]:
        """
        기본 이미지 정보 추출
        
        Args:
            image_path: 이미지 파일 경로
            image: 이미지 배열
            
        Returns:
            기본 정보 딕셔너리
        """
        file_stats = image_path.stat()
        
        return {
            'filename': image_path.name,
            'filepath': str(image_path.absolute()),
            'file_size_mb': round(file_stats.st_size / (1024 * 1024), 2),
            'created_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            'image_shape': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2] if len(image.shape) > 2 else 1
            },
            'dtype': str(image.dtype),
            'total_pixels': image.shape[0] * image.shape[1]
        }
    
    def extract_intensity_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지 강도 통계 추출
        
        Args:
            image: 이미지 배열
            
        Returns:
            강도 통계 딕셔너리
        """
        if len(image.shape) == 3:
            # RGB 이미지의 경우 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        return {
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'min_intensity': int(np.min(gray)),
            'max_intensity': int(np.max(gray)),
            'median_intensity': float(np.median(gray)),
            'q25_intensity': float(np.percentile(gray, 25)),
            'q75_intensity': float(np.percentile(gray, 75)),
            'dynamic_range': int(np.max(gray) - np.min(gray))
        }
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        텍스처 특징 추출 (간단한 통계적 방법)
        
        Args:
            image: 이미지 배열
            
        Returns:
            텍스처 특징 딕셔너리
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Laplacian (엣지 강도)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_strength = float(np.var(laplacian))
        
        # Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return {
            'edge_strength': edge_strength,
            'mean_gradient': float(np.mean(gradient_magnitude)),
            'texture_complexity': float(np.std(gray)),
            'contrast': float(gray.max() - gray.min())
        }
    
    def create_metadata(
        self,
        image_path: Path,
        image: np.ndarray,
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        완전한 메타데이터 생성
        
        Args:
            image_path: 이미지 파일 경로
            image: 이미지 배열
            analysis_results: 분석 결과 (선택)
            
        Returns:
            완전한 메타데이터 딕셔너리
        """
        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'basic_info': self.extract_basic_info(image_path, image),
            'exif_data': self.extract_exif_data(image_path),
            'intensity_statistics': self.extract_intensity_statistics(image),
            'texture_features': self.extract_texture_features(image)
        }
        
        if analysis_results:
            metadata['analysis_results'] = analysis_results
        
        return metadata
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        output_path: Path,
        format: str = 'json'
    ):
        """
        메타데이터를 파일로 저장
        
        Args:
            metadata: 메타데이터 딕셔너리
            output_path: 출력 파일 경로
            format: 저장 형식 ('json' 또는 'yaml')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        elif format == 'yaml':
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metadata saved to {output_path}")
    
    def load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """
        메타데이터 파일 로드
        
        Args:
            metadata_path: 메타데이터 파일 경로
            
        Returns:
            메타데이터 딕셔너리
        """
        if metadata_path.suffix == '.json':
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif metadata_path.suffix in ['.yaml', '.yml']:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {metadata_path.suffix}")
