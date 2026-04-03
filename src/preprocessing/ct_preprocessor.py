"""
CT Image Preprocessing Pipeline
Standard preprocessing for YOLO/nnU-Net tumor detection
"""

import numpy as np
from scipy.ndimage import zoom, median_filter
from typing import Tuple, Optional

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("[Warning] pydicom not available. Install with: pip install pydicom")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Warning] opencv-python not available. Install with: pip install opencv-python")


class CTPreprocessor:
    """
    CT 이미지 전처리 파이프라인
    
    Pipeline:
    1. DICOM → HU conversion
    2. Windowing (Soft tissue)
    3. Resampling (1mm spacing)
    4. Noise reduction
    5. CLAHE enhancement
    6. Normalization
    
    Example:
        >>> preprocessor = CTPreprocessor()
        >>> image = preprocessor.load_and_preprocess('scan.dcm')
        >>> # Returns RGB uint8 [0-255] ready for YOLO
    """
    
    def __init__(
        self,
        window_center: int = 40,
        window_width: int = 400,
        target_spacing: Optional[Tuple[float, float]] = (1.0, 1.0),
        target_size: Optional[Tuple[int, int]] = (512, 512),
        apply_clahe: bool = True,
        denoise: bool = True
    ):
        """
        Initialize CT Preprocessor
        
        Args:
            window_center: HU window center (default 40 for soft tissue)
            window_width: HU window width (default 400 for soft tissue)
            target_spacing: Target pixel spacing in mm (default 1.0×1.0)
            target_size: Target image size (default 512×512)
            apply_clahe: Apply CLAHE enhancement (default True)
            denoise: Apply median filter denoising (default True)
        """
        self.window_center = window_center
        self.window_width = window_width
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.denoise = denoise
        
        if not PYDICOM_AVAILABLE:
            print("[CTPreprocessor] Warning: pydicom not available")
        if not CV2_AVAILABLE and (apply_clahe or target_size):
            print("[CTPreprocessor] Warning: opencv-python not available")
    
    def load_and_preprocess(self, dicom_file) -> np.ndarray:
        """
        DICOM 파일 로드 및 전처리
        
        Args:
            dicom_file: DICOM file path or file-like object
        
        Returns:
            preprocessed_image: (H, W, 3) RGB uint8 [0-255]
        """
        # 1. Load DICOM
        if PYDICOM_AVAILABLE:
            dcm = pydicom.dcmread(dicom_file)
            pixel_array = dcm.pixel_array
            
            # 2. Convert to HU
            intercept = getattr(dcm, 'RescaleIntercept', 0)
            slope = getattr(dcm, 'RescaleSlope', 1)
            hu_image = pixel_array * slope + intercept
            
            # 3. Get pixel spacing
            pixel_spacing = getattr(dcm, 'PixelSpacing', [1.0, 1.0])
            current_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
        else:
            # Fallback: assume raw pixel data
            if isinstance(dicom_file, np.ndarray):
                hu_image = dicom_file
            else:
                raise ValueError("pydicom not available and input is not numpy array")
            current_spacing = (1.0, 1.0)
        
        # 4. Windowing
        windowed = self._window_image(hu_image)
        
        # 5. Resampling
        if self.target_spacing:
            resampled = self._resample_image(windowed, current_spacing)
        else:
            resampled = windowed
        
        # 6. Resize to target size
        if self.target_size and CV2_AVAILABLE:
            resized = cv2.resize(resampled, self.target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized = resampled
        
        # 7. Denoise
        if self.denoise:
            denoised = median_filter(resized, size=3)
        else:
            denoised = resized
        
        # 8. CLAHE enhancement
        if self.apply_clahe and CV2_AVAILABLE:
            enhanced = self._apply_clahe(denoised)
        else:
            enhanced = denoised
        
        # 9. Convert to RGB (for YOLO)
        if len(enhanced.shape) == 2:
            rgb_image = np.stack([enhanced] * 3, axis=-1)
        else:
            rgb_image = enhanced
        
        return rgb_image.astype(np.uint8)
    
    def _window_image(self, image: np.ndarray) -> np.ndarray:
        """Apply HU windowing"""
        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        
        windowed = np.clip(image, img_min, img_max)
        windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        return windowed
    
    def _resample_image(
        self,
        image: np.ndarray,
        current_spacing: Tuple[float, float]
    ) -> np.ndarray:
        """Resample image to target spacing"""
        resize_factor = [
            current_spacing[0] / self.target_spacing[0],
            current_spacing[1] / self.target_spacing[1]
        ]
        
        resampled = zoom(image, resize_factor, order=1)  # Bilinear
        return resampled
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if not CV2_AVAILABLE:
            return image
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def preprocess_numpy(self, image: np.ndarray, pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Preprocess numpy array (for non-DICOM images)
        
        Args:
            image: Numpy array (H, W)
            pixel_spacing: Pixel spacing in mm
        
        Returns:
            preprocessed_image: (H, W, 3) RGB uint8
        """
        # Normalize to HU-like range
        normalized = ((image - image.min()) / (image.max() - image.min()) * 4096).astype(np.float32)
        
        # Apply windowing
        windowed = self._window_image(normalized)
        
        # Resample
        if self.target_spacing:
            resampled = self._resample_image(windowed, pixel_spacing)
        else:
            resampled = windowed
        
        # Resize
        if self.target_size and CV2_AVAILABLE:
            resized = cv2.resize(resampled, self.target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized = resampled
        
        # Denoise
        if self.denoise:
            denoised = median_filter(resized, size=3)
        else:
            denoised = resized
        
        # CLAHE
        if self.apply_clahe and CV2_AVAILABLE:
            enhanced = self._apply_clahe(denoised)
        else:
            enhanced = denoised
        
        # To RGB
        rgb_image = np.stack([enhanced] * 3, axis=-1)
        
        return rgb_image.astype(np.uint8)


# Preset configurations
class CTPreprocessorPresets:
    """Preset configurations for different tissue types"""
    
    @staticmethod
    def soft_tissue():
        """Soft tissue window (abdomen, colon)"""
        return CTPreprocessor(window_center=40, window_width=400)
    
    @staticmethod
    def lung():
        """Lung window"""
        return CTPreprocessor(window_center=-600, window_width=1500)
    
    @staticmethod
    def bone():
        """Bone window"""
        return CTPreprocessor(window_center=300, window_width=1500)
    
    @staticmethod
    def brain():
        """Brain window"""
        return CTPreprocessor(window_center=40, window_width=80)
